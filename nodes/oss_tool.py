# -*- coding: utf-8 -*-

import oss2


def oss_upload(
        file,
        filename: str,
        endpoint: str,
        bucket: str,
        oss_access_key_id: str,
        oss_access_key_secret: str,
        oss_session_token: str = None,
        is_cname: bool = False,
        visit_endpoint: str = None,
        sign: bool = False,
        timeout: int = 86400
):
    """
     上传文件到阿里云OSS存储，并可选地生成签名URL。

     :param file: 要上传的本地文件路径。
     :param filename: 存储在OSS上的文件名，包含扩展名。
     :param endpoint: OSS服务的终端节点。
     :param bucket: OSS的存储桶名称。
     :param oss_access_key_id: OSS访问密钥ID。
     :param oss_access_key_secret: OSS访问密钥秘密。
     :param oss_session_token: (可选) OSS会话令牌，用于临时授权。
     :param is_cname: (可选) 是否使用CNAME方式访问OSS。
     :param visit_endpoint: (可选) 用于生成URL的访问终端节点。
     :param sign: (可选) 是否生成签名URL。
     :param timeout: (可选) 签名URL的有效期，单位为秒。
     :return: 上传后的URL或错误信息。
     """
    if oss_session_token:
        auth = oss2.StsAuth(
            access_key_id=oss_access_key_id,
            access_key_secret=oss_access_key_secret,
            security_token=oss_session_token
        )
    else:
        auth = oss2.Auth(
            access_key_id=oss_access_key_id,
            access_key_secret=oss_access_key_secret
        )

    if not endpoint.startswith('http'):
        endpoint = 'https://' + endpoint

    if visit_endpoint and not visit_endpoint.startswith('https'):
        visit_endpoint = 'https://' + visit_endpoint

    bucket_obj = oss2.Bucket(auth=auth, endpoint=endpoint, bucket_name=bucket, is_cname=is_cname)
    response = bucket_obj.put_object_from_file(filename, file)

    if response.status == 200:
        if sign:
            image_url = bucket_obj.sign_url('GET', filename, timeout)
            if visit_endpoint:
                image_url = visit_endpoint + '/' + ''.join(image_url.split('/')[3:])

        else:
            if visit_endpoint:
                image_url = visit_endpoint.rstrip('/') + '/' + filename
            else:
                endpoint_split = endpoint.split('//')
                image_url = endpoint_split[0] + '//' + bucket + '.' + endpoint_split[1] + '/' + filename
        return image_url
    else:
        print(f'status: {response.status}, failed to upload image to OSS storage')
